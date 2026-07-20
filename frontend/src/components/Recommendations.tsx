import { useCallback, useEffect, useState } from 'react'
import { addInteraction, getRecommendations } from '../api'
import type { InteractionKind, Recommendations as Recs } from '../types'

interface Props {
  userId: string
  onRestart: () => void
}

const ROMAN = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X']

export function Recommendations({ userId, onRestart }: Props) {
  const [recs, setRecs] = useState<Recs | null>(null)
  const [feedback, setFeedback] = useState<Record<number, InteractionKind>>({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const load = useCallback(() => {
    setLoading(true)
    setError('')
    setFeedback({})
    getRecommendations(userId)
      .then(setRecs)
      .catch((e) => setError(e instanceof Error ? e.message : 'Не удалось получить рекомендации'))
      .finally(() => setLoading(false))
  }, [userId])

  useEffect(load, [load])

  const sendFeedback = async (gameId: number, kind: InteractionKind) => {
    setFeedback((f) => ({ ...f, [gameId]: kind }))
    try {
      await addInteraction(userId, gameId, kind)
    } catch {
      setFeedback((f) => {
        const next = { ...f }
        delete next[gameId]
        return next
      })
    }
  }

  const feedbackGiven = Object.keys(feedback).length > 0

  if (loading) return <p className="hint">Подбираем игры…</p>
  if (error)
    return (
      <>
        <p className="error">{error}</p>
        <button className="ghost" onClick={onRestart}>Назад к настройкам</button>
      </>
    )
  if (!recs) return null

  return (
    <>
      <div className="quest-header">
        <h2>Ваш журнал заданий: топ-{recs.items.length}</h2>
        <span className="model-badge">
          модель: {recs.model_name} {recs.model_version}
        </span>
      </div>
      <p className="hint">
        Оценивайте рекомендации — «интересно» и «не интересно» учитываются в следующей выдаче.
      </p>

      <div className="quest">
        {recs.items.map((item) => (
          <article key={item.game.id} className="card">
            <div className="card-top">
              <h3>
                <span className="rank">{ROMAN[item.rank - 1] ?? item.rank}</span>
                {item.game.title}
              </h3>
              <span className={`price-tag ${item.game.price === 0 ? 'free' : ''}`}>
                {item.game.price === 0 ? 'бесплатно' : `$${item.game.price}`}
              </span>
            </div>
            <p className="desc">{item.game.description}</p>
            <div className="tags">
              {item.game.tags.slice(0, 6).map((t) => (
                <span key={t} className="tag">{t}</span>
              ))}
            </div>
            <p className="reason">{item.reason}</p>
            <div className="feedback">
              {(
                [
                  ['like', 'Интересно'],
                  ['dislike', 'Не интересно'],
                  ['played', 'Уже играл'],
                ] as [InteractionKind, string][]
              ).map(([kind, label]) => (
                <button
                  key={kind}
                  className={feedback[item.game.id] === kind ? `chosen-${kind}` : ''}
                  disabled={Boolean(feedback[item.game.id])}
                  onClick={() => sendFeedback(item.game.id, kind)}
                >
                  {label}
                </button>
              ))}
            </div>
          </article>
        ))}
      </div>

      <div className="toolbar">
        <button className="primary" disabled={!feedbackGiven} onClick={load}>
          Обновить рекомендации
        </button>
        <button className="ghost" onClick={onRestart}>Начать заново</button>
      </div>
    </>
  )
}
