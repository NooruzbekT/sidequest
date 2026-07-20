export interface Game {
  id: number
  title: string
  description: string
  tags: string[]
  price: number
  positive_ratio: number
  user_reviews: number
}

export interface User {
  id: string
  name: string | null
  genres: string[]
  max_price: number | null
  blocked_tags: string[]
}

export interface RecommendationItem {
  game: Game
  rank: number
  score: number
  reason: string
}

export interface Recommendations {
  user_id: string
  model_name: string
  model_version: string
  items: RecommendationItem[]
}

export type InteractionKind = 'like' | 'dislike' | 'played'
