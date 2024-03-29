import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";

import { useStore } from "../../store";
import { getTutorialHeatmap } from "../../api";
import { NUM_CHOICES_TUTORIAL } from "../../constants";
import { Box } from "../../components/Box/Box";
import { TileGrid } from "../../components/TileGrid/TileGrid";
import { TopBar } from "../../components/TopBar/TopBar";

const ChoicePage = (): JSX.Element => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);

  const [ehm, setEhm] = useStore((state) => [
    state.evidenceHeatmap,
    state.setEvidenceHeatmap,
  ]);
  const [thm, setThm] = useStore((state) => [
    state.trueHeatmap,
    state.setTrueHeatmap,
  ]);
  const choiceCount = useStore((state) => state.choiceCount);
  const setRandomFocusedTiles = useStore(
    (state) => state.setRandomFocusedTiles,
  );

  useEffect(() => {
    const setup = async () => {
      const hmap = await getTutorialHeatmap();
      setEhm(hmap);
      setThm(hmap);
      setRandomFocusedTiles();
      setLoading(false);
    };
    setup();
  }, []);

  useEffect(() => {
    if (choiceCount === NUM_CHOICES_TUTORIAL) {
      navigate("/tutorial/complete");
    }
  }, [choiceCount, navigate]);

  return (
    <Box className="page">
      <TopBar />
      <Box direction="row" align="flex-start" className="grids-container">
        {loading ? (
          <span>Loading...</span>
        ) : (
          <>
            <Box className="map-container">
              <Box
                direction="row"
                justify="space-between"
                className="map-title"
              >
                <span>Map</span>
                {""}
              </Box>
              <TileGrid
                heatmap={ehm}
                dynamic={false}
                tileSize={100}
                tileMargin={0}
                tileRadius={0}
              />
            </Box>
            <Box className="choice-grid-container">
              <img
                src={window.location.origin + "/assets/which-plot.png"}
                className="which-choice-image"
                alt=""
              />
              <TileGrid
                heatmap={thm}
                dynamic={true}
                revealValues={true}
                recordChoices={false}
                tileSize={130}
                tileMargin={5}
                tileRadius={20}
              />
            </Box>
          </>
        )}
      </Box>
    </Box>
  );
};

export default ChoicePage;
