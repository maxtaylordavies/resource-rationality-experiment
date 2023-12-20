import { create } from "zustand";

type Pos = {
  row: number;
  col: number;
};

export type GlobalState = {
  focusedTiles: Pos[];
  setFocusedTiles: (tiles: Pos[]) => void;
};

export const useStore = create<GlobalState>((set) => ({
  focusedTiles: [],
  setFocusedTiles: (tiles: Pos[]) => set({ focusedTiles: tiles }),
}));
