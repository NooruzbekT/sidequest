import { useEffect, useState } from 'react'
import { addInteraction, createUser, getDemoUsers, searchGames, setPreferences } from '../api'
import type { Game, User } from '../types'

const GENRES = [
  'Indie', 'Action', 'Adventure', 'RPG', 'Strategy',
  'Simulation', 'Casual', 'Puzzle', 'Multiplayer', 'Story Rich',
]
const STOP_TAGS = ['Horror', 'Violent', 'Gore', 'Sexual Content', 'Sports', 'Free to Play']
const MIN_LIKED = 5

interface Props {
  onReady: (userId: string) => void
}

export function Onboarding({ onReady }: Props) {
  const [demoUsers, setDemoUsers] = useState<User[]>([])
  const [name, setName] = useState('')
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<Game[]>([])
  const [liked, setLiked] = useState<Game[]>([])
  const [genres, setGenres] = useState<string[]>([])
  const [maxPrice, setMaxPrice] = useState(25)
  const [blocked, setBlocked] = useState<string[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    getDemoUsers().then(setDemoUsers).catch(() => setDemoUsers([]))
  }, [])

  useEffect(() => {
    if (!query.trim()) {
      setResults([])
      return
    }
    const t = setTimeout(() => {
      searchGames(query).then(setResults).catch(() => setResults([]))
    }, 250)
    return () => clearTimeout(t)
  }, [query])

  const toggle = (list: string[], value: string, set: (v: string[]) => void) =>
    set(list.includes(value) ? list.filter((x) => x !== value) : [...list, value])

  const addGame = (game: Game) => {
    if (!liked.some((g) => g.id === game.id)) setLiked([...liked, game])
    setQuery('')
    setResults([])
  }

  const canSubmit = liked.length >= MIN_LIKED && blocked.length >= 1 && !busy

  const submit = async () => {
    setBusy(true)
    setError('')
    try {
      const user = await createUser(name.trim() || null)
      await setPreferences(user.id, { genres, max_price: maxPrice, blocked_tags: blocked })
      for (const game of liked) {
        await addInteraction(user.id, game.id, 'like')
      }
      onReady(user.id)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Что-то пошло не так — попробуйте ещё раз')
      setBusy(false)
    }
  }

  return (
    <>
      {demoUsers.length > 0 && (
        <section className="panel">
          <h2>Быстрый старт</h2>
          <p className="hint">Готовые профили с предпочтениями и историей лайков.</p>
          <div className="chips">
            {demoUsers.map((u) => (
              <button key={u.id} className="chip" onClick={() => onReady(u.id)}>
                {u.name?.replace('Demo: ', '')}
              </button>
            ))}
          </div>
        </section>
      )}

      <section className="panel">
        <h2>Любимые игры</h2>
        <p className="hint">
          Отметьте минимум {MIN_LIKED} игр, которые вам понравились ({liked.length}/{MIN_LIKED}).
        </p>
        <input
          type="text"
          placeholder="Найти игру: hades, witcher, portal…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        {results.length > 0 && (
          <div className="search-results">
            {results.map((g) => (
              <button key={g.id} className="search-item" onClick={() => addGame(g)}>
                <span>{g.title}</span>
                <span className="price">{g.price === 0 ? 'бесплатно' : `$${g.price}`}</span>
              </button>
            ))}
          </div>
        )}
        {liked.length > 0 && (
          <div className="chips picked">
            {liked.map((g) => (
              <button
                key={g.id}
                className="chip on"
                title="Убрать"
                onClick={() => setLiked(liked.filter((x) => x.id !== g.id))}
              >
                {g.title} ✕
              </button>
            ))}
          </div>
        )}
      </section>

      <section className="panel">
        <h2>Жанры</h2>
        <p className="hint">Что вам обычно заходит.</p>
        <div className="chips">
          {GENRES.map((g) => (
            <button
              key={g}
              className={`chip ${genres.includes(g) ? 'on' : ''}`}
              onClick={() => toggle(genres, g, setGenres)}
            >
              {g}
            </button>
          ))}
        </div>
      </section>

      <section className="panel">
        <h2>Бюджет</h2>
        <p className="hint">Максимальная цена игры: ${maxPrice}</p>
        <input
          type="range"
          min={0}
          max={60}
          step={5}
          value={maxPrice}
          onChange={(e) => setMaxPrice(Number(e.target.value))}
        />
      </section>

      <section className="panel">
        <h2>Стоп-теги</h2>
        <p className="hint">Что показывать не нужно (минимум один).</p>
        <div className="chips">
          {STOP_TAGS.map((t) => (
            <button
              key={t}
              className={`chip blocked ${blocked.includes(t) ? 'on' : ''}`}
              onClick={() => toggle(blocked, t, setBlocked)}
            >
              {t}
            </button>
          ))}
        </div>
      </section>

      <section className="panel">
        <h2>Имя (необязательно)</h2>
        <input
          type="text"
          placeholder="Как вас называть"
          value={name}
          onChange={(e) => setName(e.target.value)}
        />
      </section>

      <button className="primary" disabled={!canSubmit} onClick={submit}>
        {busy ? 'Собираем рекомендации…' : 'Получить рекомендации'}
      </button>
      {error && <p className="error">{error}</p>}
    </>
  )
}
