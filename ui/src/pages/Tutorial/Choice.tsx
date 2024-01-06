import React, { useEffect } from "react";
import { useNavigate } from "react-router-dom";

import { useStore } from "../../store";
import { NUM_CHOICES } from "../../constants";
import { Box } from "../../components/Box/Box";
import { TileGrid } from "../../components/TileGrid/TileGrid";
import { TopBar } from "../../components/TopBar/TopBar";

const ChoicePage = (): JSX.Element => {
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
      navigate("/tutorial/complete");
      resetChoiceCount();
    }
  }, [choiceCount]);

  return (
    <Box className="page">
      <TopBar numChoices={NUM_CHOICES} phase="Tutorial" />
      <div className="choice-grid-container">
        <img
          src={window.location.origin + "/assets/which-plot.png"}
          className="which-choice-image"
        />
        <TileGrid
          heatmap={heatmap}
          dynamic={true}
          revealValues={true}
          recordChoices={false}
          tileSize={130}
          tileMargin={5}
          tileRadius={20}
        />
      </div>
    </Box>
  );
};

export default ChoicePage;
