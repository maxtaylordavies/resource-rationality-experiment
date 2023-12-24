import React, { useEffect } from "react";
import { useNavigate } from "react-router-dom";

import { useStore } from "../store";
import { NUM_CHOICES } from "../constants";
import { TileGrid } from "../components/TileGrid/TileGrid";

const TestPage = (): JSX.Element => {
  const navigate = useNavigate();

  const choiceCount = useStore((state) => state.choiceCount);
  const resetChoiceCount = useStore((state) => state.resetChoiceCount);
  const setRandomFocusedTiles = useStore(
    (state) => state.setRandomFocusedTiles
  );

  useEffect(() => {
    setRandomFocusedTiles();
  }, []);

  useEffect(() => {
    if (choiceCount === NUM_CHOICES) {
      navigate("/evidence2");
      resetChoiceCount();
    }
  }, [choiceCount]);

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
          dynamic={true}
          tileSize={60}
          revealValues={true}
          recordChoices={false}
        />
      </div>
    </div>
  );
};

export default TestPage;
