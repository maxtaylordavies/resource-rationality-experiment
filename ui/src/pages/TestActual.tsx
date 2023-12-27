import React, { useEffect } from "react";
import { useNavigate } from "react-router-dom";

import { useStore } from "../store";
import { NUM_CHOICES } from "../constants";
import { TileGrid } from "../components/TileGrid/TileGrid";
import { TopBar } from "../components/TopBar/TopBar";

const TestActual = (): JSX.Element => {
  const navigate = useNavigate();

  const heatmap = useStore((state) => state.heatmap);
  const choiceCount = useStore((state) => state.choiceCount);
  const setRandomFocusedTiles = useStore(
    (state) => state.setRandomFocusedTiles
  );

  useEffect(() => {
    setRandomFocusedTiles();
  }, []);

  useEffect(() => {
    if (choiceCount === NUM_CHOICES) {
      navigate("/complete");
    }
  }, [choiceCount]);

  return (
    <div className="page">
      <TopBar numChoices={NUM_CHOICES} />
      <img
        src={window.location.origin + "/assets/which-choose.png"}
        className="which-choose-image"
      />
      <TileGrid
        heatmap={heatmap}
        dynamic={true}
        revealValues={false}
        recordChoices={true}
      />
    </div>
  );
};

export default TestActual;
