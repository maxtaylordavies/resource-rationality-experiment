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
  p1: Pos,
  p2: Pos,
  heatmap: Heatmap,
  beta: number
): Pos => {
  const prob1 = Math.exp(heatmap[p1.row][p1.col] / beta);
  const prob2 = Math.exp(heatmap[p2.row][p2.col] / beta);
  const rand = Math.random();
  if (rand < prob1 / (prob1 + prob2)) {
    return p1;
  } else {
    return p2;
  }
};
