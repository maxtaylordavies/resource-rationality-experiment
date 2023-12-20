import React from "react";
import { motion } from "framer-motion";

import { useStore } from "../../store";
import "./tile-grid.css";

type TileGridProps = {
  rows: number;
  cols: number;
};

export const TileGrid = ({ rows, cols }: TileGridProps) => {
  const focusedTiles = useStore((state) => state.focusedTiles);

  const isFocused = (row: number, col: number) => {
    return focusedTiles.some((tile) => tile.row === row && tile.col === col);
  };

  return (
    <div className="tile-grid">
      {Array.from(Array(rows).keys()).map((row, i) => {
        return (
          <div className="tile-grid-row" key={row}>
            {Array.from(Array(cols).keys()).map((col, j) => {
              return <Tile key={col} focused={isFocused(i, j)} />;
            })}
          </div>
        );
      })}
    </div>
  );
};

type TileProps = {
  key: number;
  focused?: boolean;
};

const Tile = ({ key, focused }: TileProps) => {
  return (
    <motion.div
      key={key}
      className="tile-grid-tile"
      initial={{ scale: 1.0, opacity: 0.5 }}
      animate={{ scale: focused ? 1.15 : 1.0, opacity: focused ? 1.0 : 0.5 }}
      whileHover={{ scale: focused ? 1.25 : 1.0 }}
      style={{ pointerEvents: focused ? "auto" : "none", cursor: "pointer" }}
    ></motion.div>
  );
};
