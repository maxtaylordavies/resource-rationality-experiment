import { Pos, Heatmap } from "./types";

const PREFIX = "_rre_";

export const getValueFromUrlOrLocalstorage = (key: string) => {
  const params = new URLSearchParams(window.location.search);
  return (
    params.get(key) ||
    JSON.parse(localStorage.getItem(`${PREFIX}${key}`) || "null")
  );
};

export const getProlificMetadata = () => {
  const metadata: { [key: string]: string } = {};
  const params = new URLSearchParams(window.location.search);
  params.forEach((val, key) => {
    if (key.startsWith("PRLFC")) {
      metadata[key] = val;
    }
  });
  return metadata;
};

export const writeToLocalStorage = (key: string, val: any) => {
  localStorage.setItem(`${PREFIX}${key}`, JSON.stringify(val));
};

export const removeFromLocalStorage = (key: string) => {
  localStorage.removeItem(`${PREFIX}${key}`);
};

export const sampleRandomChoicePair = (
  sideLength: number,
  minDist = 2
): Pos[] => {
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

export const chooseTile = (
  tiles: Pos[],
  heatmap: Heatmap,
  beta: number
): number => {
  let probs = tiles.map((tile) => Math.exp(heatmap[tile.row][tile.col] / beta));
  const sum = probs.reduce((a, b) => a + b, 0);
  probs = probs.map((v) => v / sum);

  const r = Math.random();
  let acc = 0;
  for (let i = 0; i < probs.length; i++) {
    acc += probs[i];
    if (r < acc) {
      return i;
    }
  }

  return -1;
};
