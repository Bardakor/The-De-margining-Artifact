import type { Market, ScorelineMatrix } from "../types.js";
import { sumWhere, toFairOdds } from "./matchOdds.js";

export function bttsMarket(matrix: ScorelineMatrix): Market {
  const yes = sumWhere(matrix, (h, a) => h > 0 && a > 0);
  const no = 1 - yes;
  return {
    key: "BTTS",
    label: "Both Teams To Score",
    selections: [
      { key: "BTTS:YES", label: "Yes", probability: yes, fairOdds: toFairOdds(yes) },
      { key: "BTTS:NO", label: "No", probability: no, fairOdds: toFairOdds(no) },
    ],
  };
}
