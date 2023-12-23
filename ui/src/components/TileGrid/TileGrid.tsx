import React from "react";
import { motion } from "framer-motion";

import { COLORS } from "../../constants";
import { useStore } from "../../store";
import "./tile-grid.css";

type TileGridProps = {
  rows: number;
  cols: number;
  dynamic: boolean;
  heatmap?: number[][];
};

export const TileGrid = ({ rows, cols, dynamic, heatmap }: TileGridProps) => {
  console.log("heatmap", heatmap);

  const focusedTiles = useStore((state) => state.focusedTiles);
  const setRandomFocusedTiles = useStore(
    (state) => state.setRandomFocusedTiles
  );
  const incrementChoiceCount = useStore((state) => state.incrementChoiceCount);

  const tileValue = (row: number, col: number) => {
    if (!heatmap || heatmap.length === 0) return -1;
    return heatmap[row][col];
  };

  const isFocused = (row: number, col: number) => {
    return (
      dynamic &&
      focusedTiles.some((tile) => tile.row === row && tile.col === col)
    );
  };

  const onTileClick = () => {
    if (!dynamic) return;
    incrementChoiceCount();
    setRandomFocusedTiles();
  };

  return (
    <div className="tile-grid">
      {Array.from(Array(rows).keys()).map((row, i) => {
        return (
          <div className="tile-grid-row" key={row}>
            {Array.from(Array(cols).keys()).map((col, j) => {
              let type: TileType = "normal";
              if (dynamic) {
                type = isFocused(i, j) ? "focused" : "unfocused";
              }
              return (
                <Tile
                  key={col}
                  value={tileValue(i, j)}
                  type={type}
                  onClick={onTileClick}
                />
              );
            })}
          </div>
        );
      })}
    </div>
  );
};

type TileType = "focused" | "unfocused" | "normal";

type TileProps = {
  key: number;
  value: number;
  type?: TileType;
  onClick: () => void;
};

const Tile = ({ key, value, type, onClick }: TileProps) => {
  const color = value === -1 ? "grey" : COLORS[value];

  return (
    <motion.div
      key={key}
      className="tile-grid-tile"
      initial={{ scale: 1.0, opacity: 0.5 }}
      animate={{
        scale: type === "focused" ? 1.1 : 1.0,
        opacity: type === "unfocused" ? 0.5 : 1.0,
      }}
      whileHover={{ scale: type === "focused" ? 1.2 : 1.0 }}
      style={{
        backgroundColor: color,
        pointerEvents: type === "focused" ? "auto" : "none",
        cursor: "pointer",
      }}
      onClick={onClick}
    ></motion.div>
  );
};
