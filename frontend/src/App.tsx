/**
 * Main App component for SLB Scenario Planner.
 *
 * This is the entry point for the frontend application.
 * PR6 will add the full ScenarioPlannerPage with components.
 */

import './App.css'

function App() {
  return (
    <div className="app">
      <header className="app-header">
        <h1>SLB Scenario Planner</h1>
        <p className="subtitle">Sale-Leaseback Funding Program Analyzer</p>
      </header>

      <main className="app-main">
        <div className="placeholder-content">
          <h2>Frontend Setup Complete</h2>
          <p>
            The frontend scaffolding is in place. Components will be added in PR6.
          </p>
          <div className="feature-list">
            <h3>Coming Features:</h3>
            <ul>
              <li>Scenario input form</li>
              <li>Run list sidebar</li>
              <li>Audit trace timeline</li>
              <li>Numeric invariants display</li>
              <li>Metrics visualization</li>
              <li>Asset selection table</li>
            </ul>
          </div>
          <div className="tech-stack">
            <h3>Tech Stack:</h3>
            <ul>
              <li>React 19 + TypeScript</li>
              <li>Vite build tool</li>
              <li>FastAPI backend</li>
            </ul>
          </div>
        </div>
      </main>

      <footer className="app-footer">
        <p>SLB Agent • Auquan</p>
      </footer>
    </div>
  )
}

export default App
