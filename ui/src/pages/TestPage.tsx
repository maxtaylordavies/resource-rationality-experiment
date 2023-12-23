import React, { useEffect } from "react";

import { useStore } from "../store";
import { GRID_SIZE, NUM_CHOICES } from "../constants";
import { TileGrid } from "../components/TileGrid/TileGrid";

const TestPage = (): JSX.Element => {
  const choiceCount = useStore((state) => state.choiceCount);
  const setRandomFocusedTiles = useStore(
    (state) => state.setRandomFocusedTiles
  );

  useEffect(() => {
    setRandomFocusedTiles();
  }, []);

  return (
    <div className="page">
      <div className="test-grid-container">
        <div className="choice-count-box">
          <span>Completed:</span>
          <span className="choice-count">
            {choiceCount}/{NUM_CHOICES}
          </span>
        </div>
        <TileGrid
          rows={GRID_SIZE}
          cols={GRID_SIZE}
          dynamic={true}
          heatmap={[]}
        />
      </div>
    </div>
  );
};

export default TestPage;
