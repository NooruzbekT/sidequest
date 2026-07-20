import type { Game, InteractionKind, Recommendations, User } from './types'

const BASE = '/api'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(BASE + path, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!resp.ok) {
    const body = await resp.json().catch(() => null)
    throw new Error(body?.detail ?? `Ошибка сервера (${resp.status})`)
  }
  return resp.json()
}

export const searchGames = (query: string, limit = 8) =>
  request<Game[]>(`/games?query=${encodeURIComponent(query)}&limit=${limit}`)

export const getDemoUsers = () => request<User[]>('/users/demo')

export const createUser = (name: string | null) =>
  request<User>('/users', { method: 'POST', body: JSON.stringify({ name }) })

export const setPreferences = (
  userId: string,
  prefs: { genres: string[]; max_price: number | null; blocked_tags: string[] },
) =>
  request<User>(`/users/${userId}/preferences`, {
    method: 'POST',
    body: JSON.stringify(prefs),
  })

export const addInteraction = (userId: string, gameId: number, kind: InteractionKind) =>
  request(`/users/${userId}/interactions`, {
    method: 'POST',
    body: JSON.stringify({ game_id: gameId, kind }),
  })

export const getRecommendations = (userId: string) =>
  request<Recommendations>(`/users/${userId}/recommendations`)
