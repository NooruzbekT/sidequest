import { useState } from 'react'
import { Onboarding } from './components/Onboarding'
import { Recommendations } from './components/Recommendations'

export default function App() {
  const [userId, setUserId] = useState<string | null>(null)

  return (
    <main>
      <h1>
        Side<span className="accent">Quest</span>
      </h1>
      <p className="subtitle">Подбирает следующую игру под ваши вкусы, бюджет и стоп-теги.</p>
      {userId ? (
        <Recommendations userId={userId} onRestart={() => setUserId(null)} />
      ) : (
        <Onboarding onReady={setUserId} />
      )}
    </main>
  )
}
