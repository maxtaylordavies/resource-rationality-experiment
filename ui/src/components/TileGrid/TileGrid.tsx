import React from "react";

import "./tile-grid.css";

type TileGridProps = {
  rows: number;
  cols: number;
};

export const TileGrid = ({ rows, cols }: TileGridProps) => {
  return (
    <div className="tile-grid">
      {Array.from(Array(rows).keys()).map((row) => {
        return (
          <div className="tile-grid-row" key={row}>
            {Array.from(Array(cols).keys()).map((col) => {
              return <Tile key={col} />;
            })}
          </div>
        );
      })}
    </div>
  );
};

const Tile = () => {
  return <div className="tile-grid-tile"></div>;
};
