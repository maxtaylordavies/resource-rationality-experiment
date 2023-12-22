import React, { useEffect } from "react";

import { useStore } from "../store";
import { GRID_SIZE } from "../constants";
import { TileGrid } from "../components/TileGrid/TileGrid";

const TestPage = (): JSX.Element => {
  const setRandomFocusedTiles = useStore(
    (state) => state.setRandomFocusedTiles
  );

  useEffect(() => {
    setRandomFocusedTiles();
  }, []);

  return (
    <div className="App">
      <TileGrid rows={GRID_SIZE} cols={GRID_SIZE} dynamic={true} heatmap={[]} />
    </div>
  );
};

export default TestPage;
