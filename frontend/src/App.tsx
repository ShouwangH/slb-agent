/**
 * Main App component for SLB Scenario Planner.
 *
 * This is the entry point for the frontend application.
 */

import "./App.css";
import { ErrorBoundary } from "./components";
import { ScenarioPlannerPage } from "./pages";

function App() {
  return (
    <div className="app">
      <header className="app-header">
        <h1>SLB Scenario Planner</h1>
        <p className="subtitle">Sale-Leaseback Funding Program Analyzer</p>
      </header>

      <ErrorBoundary>
        <ScenarioPlannerPage />
      </ErrorBoundary>

      <footer className="app-footer">
        <p>SLB Agent • Auquan</p>
      </footer>
    </div>
  );
}

export default App;
