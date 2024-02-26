import { create } from "zustand";

import { Session, Heatmap, Pos } from "./types";
import { INITIAL_SCORE } from "./constants";
import { sampleRandomChoicePair } from "./utils";

export type GlobalState = {
  session: Session | null;
  setSession: (session: Session) => void;
  heatmap: Heatmap;
  setHeatmap: (heatmap: Heatmap) => void;
  evidenceHeatmap: Heatmap;
  setEvidenceHeatmap: (heatmap: Heatmap) => void;
  trueHeatmap: Heatmap;
  setTrueHeatmap: (heatmap: Heatmap) => void;
  round: number;
  incrementRound: () => void;
  chosenPatchSize: number;
  setChosenPatchSize: (ps: number) => void;
  score: number;
  incrementScore: (amount: number) => void;
  resetScore: () => void;
  focusedTiles: Pos[];
  setFocusedTiles: (tiles: Pos[]) => void;
  setRandomFocusedTiles: () => void;
  choiceCount: number;
  incrementChoiceCount: () => void;
  resetChoiceCount: () => void;
};

export const useStore = create<GlobalState>((set) => ({
  session: null,
  setSession: (session: Session) => set({ session: session }),
  heatmap: [],
  setHeatmap: (heatmap: Heatmap) => set({ heatmap: heatmap }),
  evidenceHeatmap: [],
  setEvidenceHeatmap: (heatmap: Heatmap) => set({ evidenceHeatmap: heatmap }),
  trueHeatmap: [],
  setTrueHeatmap: (heatmap: Heatmap) => set({ trueHeatmap: heatmap }),
  round: 0,
  incrementRound: () =>
    set((state) => ({
      round: state.round + 1,
      choiceCount: 0,
      chosenPatchSize: -1,
    })),
  chosenPatchSize: -1,
  setChosenPatchSize: (ps: number) => set({ chosenPatchSize: ps }),
  score: INITIAL_SCORE,
  incrementScore: (amount: number) =>
    set((state) => ({ score: state.score + amount })),
  resetScore: () => set({ score: INITIAL_SCORE }),
  focusedTiles: [],
  setFocusedTiles: (tiles: Pos[]) => set({ focusedTiles: tiles }),
  setRandomFocusedTiles: () =>
    set((state) => ({
      focusedTiles: sampleRandomChoicePair(state.trueHeatmap.length) as Pos[],
    })),
  choiceCount: 0,
  incrementChoiceCount: () =>
    set((state) => ({ choiceCount: state.choiceCount + 1 })),
  resetChoiceCount: () => set({ choiceCount: 0 }),
}));
