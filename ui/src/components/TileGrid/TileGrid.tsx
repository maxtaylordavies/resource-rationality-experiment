import { useState, useEffect } from "react";
import { motion } from "framer-motion";

import { COLORS, CHOICE_REWARD } from "../../constants";
import { Heatmap, useStore } from "../../store";
import { recordChoiceResult } from "../../api";
import "./tile-grid.css";

type TileGridProps = {
  heatmap: Heatmap;
  dynamic: boolean;
  revealValues?: boolean;
  recordChoices?: boolean;
  tileSize?: number;
  tileMargin?: number;
  tileRadius?: number;
};

export const TileGrid = ({
  heatmap,
  dynamic,
  revealValues,
  recordChoices,
  tileSize = 60,
  tileMargin = 3,
  tileRadius = 10,
}: TileGridProps) => {
  const session = useStore((state) => state.session);
  const round = useStore((state) => state.round);
  const patchSize = useStore((state) => state.chosenPatchSize);
  const focusedTiles = useStore((state) => state.focusedTiles);
  const setRandomFocusedTiles = useStore(
    (state) => state.setRandomFocusedTiles,
  );
  const incrementScore = useStore((state) => state.incrementScore);
  const incrementChoiceCount = useStore((state) => state.incrementChoiceCount);
  const [colors, setColors] = useState<string[][]>([]);

  const initialiseTileColors = () => {
    const _colors: string[][] = [];
    for (let i = 0; i < heatmap.length; i++) {
      _colors.push([]);
      for (let j = 0; j < heatmap.length; j++) {
        _colors[i].push(dynamic ? "grey" : COLORS[heatmap[i][j]]);
      }
    }
    return _colors;
  };

  useEffect(() => {
    if (heatmap.length > 0) {
      setColors(initialiseTileColors());
    }
  }, [heatmap]);

  const isFocused = (row: number, col: number) => {
    return (
      dynamic &&
      focusedTiles.some((tile) => tile.row === row && tile.col === col)
    );
  };

  const updateColors = (
    tiles: { row: number; col: number }[],
    _colors: string[],
  ) => {
    const newColors = [...colors];
    tiles.forEach((tile, idx) => {
      newColors[tile.row][tile.col] = _colors[idx];
    });
    setColors(newColors);
  };

  const onTileClick = async (row: number, col: number) => {
    if (!dynamic || session === null) {
      return;
    }

    const selected = focusedTiles.findIndex(
      (tile) => tile.row === row && tile.col === col,
    );
    const values = focusedTiles.map((tile) => heatmap[tile.row][tile.col]);
    if (values[selected] === Math.max(...values)) {
      incrementScore(CHOICE_REWARD);
    } else {
      incrementScore(-CHOICE_REWARD);
    }

    if (recordChoices) {
      await recordChoiceResult(session.id, round, patchSize, {
        choice: focusedTiles,
        selected,
      });
    }

    if (revealValues) {
      updateColors(
        focusedTiles,
        focusedTiles.map((tile) => COLORS[heatmap[tile.row][tile.col]]),
      );
      setTimeout(() => {
        updateColors(
          focusedTiles,
          focusedTiles.map(() => "grey"),
        );
        incrementChoiceCount();
        setRandomFocusedTiles();
      }, 1000);
    } else {
      incrementChoiceCount();
      setRandomFocusedTiles();
    }
  };

  return colors && heatmap.length === colors.length ? (
    <div className="tile-grid">
      {Array.from(Array(heatmap.length).keys()).map((row, i) => {
        return (
          <div className="tile-grid-row" key={row}>
            {Array.from(Array(heatmap.length).keys()).map((col, j) => {
              let type: TileType = "normal";
              if (dynamic) {
                type = isFocused(i, j) ? "focused" : "unfocused";
              }
              return (
                <Tile
                  key={col}
                  type={type}
                  color={colors[i][j]}
                  onClick={() => onTileClick(i, j)}
                  size={tileSize}
                  margin={tileMargin}
                  radius={tileRadius}
                />
              );
            })}
          </div>
        );
      })}
    </div>
  ) : (
    <div>uh oh</div>
  );
};

type TileType = "focused" | "unfocused" | "normal";

type TileProps = {
  key: number;
  type?: TileType;
  color: string;
  size?: number;
  onClick: () => void;
  margin?: number;
  radius?: number;
};

const Tile = ({
  key,
  type,
  color,
  size,
  onClick,
  margin = 3,
  radius = 10,
}: TileProps) => {
  return (
    <motion.div
      key={key}
      className="tile-grid-tile"
      initial={{ opacity: 0.5 }}
      animate={{
        opacity: type === "unfocused" ? 0.5 : 1.0,
      }}
      whileHover={{ scale: type === "focused" ? 1.07 : 1.0 }}
      style={{
        backgroundColor: color,
        pointerEvents: type === "focused" ? "auto" : "none",
        cursor: "pointer",
        width: size,
        height: size,
        margin: margin,
        borderRadius: radius,
      }}
      onClick={onClick}
    />
  );
};
