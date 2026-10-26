import { Component, type ReactNode } from 'react'

type State = { error: Error | null }

/** Keeps one malformed result from blanking the whole console. */
export class ErrorBoundary extends Component<{ children: ReactNode }, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  render() {
    if (!this.state.error) return this.props.children
    return (
      <div className="panel-body" role="alert">
        <p className="empty">
          This view hit an unexpected result and stopped: {this.state.error.message}. Reload the
          page to start again; saved runs open from their link.
        </p>
      </div>
    )
  }
}
