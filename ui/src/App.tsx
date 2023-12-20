import React from "react";

import { TileGrid } from "./components/TileGrid/TileGrid";
import "./App.css";

const App = () => {
  return (
    <div className="App">
      <TileGrid rows={4} cols={4} />
    </div>
  );
};

export default App;
