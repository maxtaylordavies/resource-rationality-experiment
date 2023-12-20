import React, { useEffect } from "react";

import { useStore } from "./store";
import { TileGrid } from "./components/TileGrid/TileGrid";
import "./App.css";

const App = () => {
  const setFocusedTiles = useStore((state) => state.setFocusedTiles);

  useEffect(() => {
    setFocusedTiles([
      { row: 0, col: 0 },
      { row: 2, col: 2 },
    ]);
  }, []);

  return (
    <div className="App">
      <TileGrid rows={4} cols={4} />
    </div>
  );
};

export default App;
