# SideQuest — персональный рекомендатор игр

Веб-сервис, который помогает выбрать следующую игру. Пользователь отмечает любимые игры,
жанры, бюджет и стоп-теги — сервис выдаёт топ-10 персональных рекомендаций с объяснением
каждой и учится на оценках «интересно» / «не интересно».

Под капотом — не «чат с LLM», а воспроизводимый ML-пайплайн: три подхода к рекомендациям
(popularity baseline, content-based, collaborative), offline-evaluation с временным сплитом,
fallback при сбое модели, фоновое переобучение и метрики.

> Статус: **День 3 из 6** — backend с baseline-выдачей и React-интерфейсом, воспроизводимый
> offline-evaluation, две модели с честным сравнением.

## Архитектура

```mermaid
flowchart TB
    UI["Web UI<br/>React + Vite"]
    API["Backend API<br/>FastAPI"]
    PG[("PostgreSQL")]
    REDIS[("Redis / RQ")]
    WORKER["Worker<br/>импорт + обучение"]
    MODELS["Model artifacts"]

    UI -->|HTTP/JSON| API
    API --> PG
    API --> REDIS
    REDIS --> WORKER
    WORKER --> PG
    WORKER --> MODELS
    API --> MODELS
```

Подробнее — [docs/architecture.md](docs/architecture.md).

## Технологии

| Слой | Выбор | Почему |
|---|---|---|
| Backend | Python 3.13, FastAPI, SQLAlchemy 2, Alembic | рекомендованный стек ТЗ; типизация через Pydantic |
| БД | PostgreSQL 17 | реляционная модель сущностей, миграции |
| Очередь | Redis + RQ | две редкие фоновые задачи — RQ минимален и объясним; worker живёт в Docker |
| ML | scikit-learn, scipy | классические подходы, воспроизводимость |
| Frontend | React + Vite | простой SPA для демонстрации сценария |

Ключевые решения и их обоснования — в [DECISIONS.md](DECISIONS.md).

## Данные

Источник: [Game Recommendations on Steam](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam)
(Kaggle, автор Anton Kozyriev).

- **Лицензия:** CC0: Public Domain. Персональных данных нет — user ID анонимизированы автором датасета.
- **Версия:** 28 (обновлён 2024-08-14). **Дата скачивания:** 2026-07-20.
- **Состав:**
  - `games.csv` — ~51k игр: название, дата релиза, рейтинг, цена;
  - `games_metadata.json` — описания и теги игр;
  - `users.csv` — ~14M пользователей (анонимные ID, счётчики);
  - `recommendations.csv` — 41M+ отзывов: user, game, is_recommended, часы, дата.
- Почему он, а не предложенный в ТЗ Steam Reviews Dataset: есть цена (фильтр по бюджету),
  теги и описания (content-based), даты (временной сплит для honest evaluation) и явная
  бинарная оценка. Обоснование — в [DECISIONS.md](DECISIONS.md).
- Сырые данные в репозиторий не коммитятся (`data/raw/` в .gitignore); в репо будет только
  маленький demo-набор. Скачивание и фиксация версии/хешей:

```bash
python ml/download_data.py   # скачает в data/raw и создаст manifest.json c sha256
```

- Результаты EDA (sparsity, распределения, топ-теги) — [docs/eda.md](docs/eda.md).
- Какие поля исключаются из обучения и почему — будет описано вместе с train/test-сплитом
  (День 3); принцип: никаких признаков, появляющихся после момента рекомендации.

## Быстрый старт

Требования: Docker + Docker Compose.

```bash
git clone <repo-url> sidequest && cd sidequest
docker compose up --build
# API: http://localhost:8000, Swagger: http://localhost:8000/docs
curl http://localhost:8000/health
```

`.env` не обязателен (в compose есть значения по умолчанию); шаблон — [.env.example](.env.example).

## Разработка

```bash
# окружение (Python 3.13)
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r backend/requirements.txt

# тесты и линт
pytest
ruff check .

# EDA по сырым данным
python ml/eda.py
```

## API

Будет доступно в Swagger (`/docs`). Реализовано на текущий момент:

| Метод | Путь | Описание |
|---|---|---|
| GET | `/health` | статус приложения и зависимостей (db, redis) |

Полный набор endpoint'ов (users, preferences, interactions, recommendations, admin) — Дни 2–5.

## Метрики моделей

Offline-evaluation на временном сплите: train — взаимодействия до 2022-01-01, test — 2022 год.
Каталог (топ-5000 игр) и порог активности пользователей (≥5 отзывов, 50k, seed 42) отбираются
**только по train-периоду** — иначе отбор подсматривает в будущее и смещает сравнение.
Параметры и sha256 исходников — в `data/processed/split_manifest.json`, артефакты — в `ml/results/`.

| Модель | P@10 | R@10 | Coverage | Diversity | Latency mean | Latency p95 |
|---|---|---|---|---|---|---|
| popularity (baseline) | 0.0075 | 0.0274 | 0.5% | 0.891 | <0.1 ms | <0.1 ms |
| content-based (TF-IDF теги+описания) | **0.0089** | **0.0335** | **48.3%** | 0.652 | 0.5 ms | 0.7 ms |

**Главная метрика — Precision@10**: продукт показывает ровно 10 карточек, и доля попаданий
в них — то, что видит пользователь. Content-based обходит baseline по P@10 (+19%) и R@10
(+22%) и персонализирует почти половину каталога, тогда как baseline крутит всем одни и
те же ~23 игры (coverage 0.5%).

**Урок методологии (воспроизводимый):** в первой версии сплита каталог отбирался по
all-time популярности — с утечкой тестового периода в дизайн эксперимента — и baseline
«выигрывал» (P@10 0.0085 vs 0.0074). После переноса отбора на train-период результат
перевернулся. Абсолютные значения P@10 при sparsity 99.994% и медиане 1 отзыв на
пользователя закономерно малы (случайная выдача дала бы ~0.0005 — модели лучше в ~15 раз).

Оговорки: eval-популяция — пользователи с ≥1 положительным отзывом в тесте; доля пустых
выдач репортится в артефактах (`empty_recs_share`, в текущем прогоне 0 у обеих моделей).

Исключённые из обучения поля и почему — в `split_manifest.json` (`excluded_fields`):
`hours` растут после момента рекомендации, `helpful/funny` появляются после публикации
отзыва, агрегаты каталога в eval заменены на train-only счётчики.

Воспроизведение:

```bash
python ml/download_data.py   # сырые данные + manifest
python ml/split.py           # train/test-сплит + split_manifest.json
python ml/evaluate.py        # метрики обеих моделей → ml/results/*.json
```

## Ограничения и следующие шаги

- День 1: нет UI, нет БД-схемы, нет моделей — только каркас, данные и здоровье сервиса.
- План: День 2 — миграции, demo-импорт, onboarding, baseline; День 3 — evaluation +
  content-based; День 4 — collaborative + feedback; День 5 — фон, fallback, тесты, CI;
  День 6 — документация, инциденты, видео.
