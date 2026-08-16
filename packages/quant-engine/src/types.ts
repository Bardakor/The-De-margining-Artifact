/** Expected goals for each side of a single match. */
export interface MatchLambdas {
  readonly home: number;
  readonly away: number;
}

/**
 * Joint distribution over final scores.
 * `cells[h][a]` is P(home scores h AND away scores a). Sums to 1.
 */
export interface ScorelineMatrix {
  readonly maxGoals: number;
  readonly cells: readonly (readonly number[])[];
}

/** One outcome within a market, with its fair (zero-margin) price. */
export interface Selection {
  readonly key: string;
  readonly label: string;
  readonly probability: number;
  readonly fairOdds: number;
}

/** A complete, mutually exclusive and exhaustive set of selections. */
export interface Market {
  readonly key: string;
  readonly label: string;
  readonly selections: readonly Selection[];
}

/** A selection whose price has had the book margin applied. */
export interface PricedSelection extends Selection {
  readonly odds: number;
}

export interface PricedMarket {
  readonly key: string;
  readonly label: string;
  readonly selections: readonly PricedSelection[];
  /** Sum of 1/odds across selections. Equals the configured overround. */
  readonly bookSum: number;
}
