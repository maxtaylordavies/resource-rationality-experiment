import React, { useEffect } from "react";
import { useNavigate } from "react-router-dom";

import { useStore } from "../store";
import { NUM_CHOICES } from "../constants";
import { TileGrid } from "../components/TileGrid/TileGrid";
import { ChoiceCounter } from "../components/ChoiceCounter/ChoiceCounter";

const TestPage = (): JSX.Element => {
  const navigate = useNavigate();

  const heatmap = useStore((state) => state.heatmap);
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
      <ChoiceCounter numChoices={NUM_CHOICES} />
      <img
        src={window.location.origin + "/assets/which-choose.png"}
        className="which-choose-image"
      />
      <TileGrid
        heatmap={heatmap}
        dynamic={true}
        tileSize={60}
        revealValues={true}
        recordChoices={false}
      />
    </div>
  );
};

export default TestPage;
