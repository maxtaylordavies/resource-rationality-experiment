import { create } from "zustand";

export type Session = {
  id: number;
  experimentId: string;
  userId: string;
  createdAt: Date;
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
  focusedTiles: Pos[];
  setFocusedTiles: (tiles: Pos[]) => void;
  setRandomFocusedTiles: () => void;
  choiceCount: number;
  incrementChoiceCount: () => void;
  resetChoiceCount: () => void;
  // choiceHistory: ChoiceResult[];
  // recordChoice: (result: ChoiceResult) => void;
};

export const useStore = create<GlobalState>((set) => ({
  session: null,
  setSession: (session: Session) => set({ session: session }),
  heatmap: [],
  setHeatmap: (heatmap: Heatmap) => set({ heatmap: heatmap }),
  focusedTiles: [],
  setFocusedTiles: (tiles: Pos[]) => set({ focusedTiles: tiles }),
  setRandomFocusedTiles: () =>
    set((state) => ({
      focusedTiles: sampleRandomChoicePair(state.heatmap.length),
    })),
  choiceCount: 0,
  incrementChoiceCount: () =>
    set((state) => ({ choiceCount: state.choiceCount + 1 })),
  resetChoiceCount: () => set({ choiceCount: 0 }),
  // choiceHistory: [],
  // recordChoice: (result: ChoiceResult) =>
  //   set((state) => ({ choiceHistory: [...state.choiceHistory, result] })),
}));
