import React, { useEffect } from "react";

import { useStore } from "./store";
import { GRID_SIZE } from "./constants";
import { getRandomHeatmap } from "./api";
import { TileGrid } from "./components/TileGrid/TileGrid";
import "./App.css";

const App = () => {
  const heatmap = useStore((state) => state.heatmap);
  const setHeatmap = useStore((state) => state.setHeatmap);
  const setRandomFocusedTiles = useStore(
    (state) => state.setRandomFocusedTiles
  );

  const doInitialSetup = async () => {
    const hmap = await getRandomHeatmap();
    setHeatmap(hmap);
    setRandomFocusedTiles();
  };

  useEffect(() => {
    doInitialSetup();
  }, []);

  return (
    <div className="App">
      <TileGrid rows={GRID_SIZE} cols={GRID_SIZE} dynamic={true} heatmap={[]} />
    </div>
  );
};

export default App;
