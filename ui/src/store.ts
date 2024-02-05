import { create } from "zustand";

import { INITIAL_SCORE } from "./constants";

export type Session = {
  id: number;
  experiment_id: string;
  user_id: string;
  created_at: Date;
  choice_reward: number;
};

export type Heatmap = number[][];

export type Pos = {
  row: number;
  col: number;
};

export type ChoiceResult = {
  choice: Pos[];
  selected: number;
};

const sampleRandomChoicePair = (sideLength: number, minDist = 2): Pos[] => {
  const randTile = () => {
    return {
      row: Math.floor(Math.random() * sideLength),
      col: Math.floor(Math.random() * sideLength),
    };
  };
  const dist = (p1: Pos, p2: Pos) => {
    return Math.abs(p2.row - p1.row) + Math.abs(p2.col - p1.col);
  };
  const tile1 = randTile();
  while (true) {
    const tile2 = randTile();
    if (dist(tile1, tile2) >= minDist) {
      return [tile1, tile2];
    }
  }
};

export type GlobalState = {
  session: Session | null;
  setSession: (session: Session) => void;
  heatmap: Heatmap;
  setHeatmap: (heatmap: Heatmap) => void;
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
};

export const useStore = create<GlobalState>((set) => ({
  session: null,
  setSession: (session: Session) => set({ session: session }),
  heatmap: [],
  setHeatmap: (heatmap: Heatmap) => set({ heatmap: heatmap }),
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
      focusedTiles: sampleRandomChoicePair(state.heatmap.length),
    })),
  choiceCount: 0,
  incrementChoiceCount: () =>
    set((state) => ({ choiceCount: state.choiceCount + 1 })),
}));
