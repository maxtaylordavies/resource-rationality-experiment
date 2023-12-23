import { create } from "zustand";

import { GRID_SIZE } from "./constants";

type Heatmap = number[][];

type Pos = {
  row: number;
  col: number;
};

const sampleRandomChoicePair = (minDist = 2): Pos[] => {
  const randTile = () => {
    return {
      row: Math.floor(Math.random() * GRID_SIZE),
      col: Math.floor(Math.random() * GRID_SIZE),
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
  heatmap: Heatmap;
  setHeatmap: (heatmap: Heatmap) => void;
  focusedTiles: Pos[];
  setFocusedTiles: (tiles: Pos[]) => void;
  setRandomFocusedTiles: () => void;
  choiceCount: number;
  incrementChoiceCount: () => void;
};

export const useStore = create<GlobalState>((set) => ({
  heatmap: [],
  setHeatmap: (heatmap: Heatmap) => set({ heatmap: heatmap }),
  focusedTiles: [],
  setFocusedTiles: (tiles: Pos[]) => set({ focusedTiles: tiles }),
  setRandomFocusedTiles: () => set({ focusedTiles: sampleRandomChoicePair() }),
  choiceCount: 0,
  incrementChoiceCount: () =>
    set((state) => ({ choiceCount: state.choiceCount + 1 })),
}));
