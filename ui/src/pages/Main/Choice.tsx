import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";

import { useStore } from "../../store";
import { NUM_ROUNDS, NUM_CHOICES, VEGETABLES } from "../../constants";
import { getHeatmapFromFile } from "../../api";
import { Box } from "../../components/Box/Box";
import { TileGrid } from "../../components/TileGrid/TileGrid";
import { TopBar } from "../../components/TopBar/TopBar";

const ChoicePage = (): JSX.Element => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);

  const session = useStore((state) => state.session);
  const round = useStore((state) => state.round);
  const [ehm, setEhm] = useStore((state) => [
    state.evidenceHeatmap,
    state.setEvidenceHeatmap,
  ]);
  const [thm, setThm] = useStore((state) => [
    state.trueHeatmap,
    state.setTrueHeatmap,
  ]);
  const chosenPatchSize = useStore((state) => state.chosenPatchSize);
  const choiceCount = useStore((state) => state.choiceCount);
  const setRandomFocusedTiles = useStore(
    (state) => state.setRandomFocusedTiles,
  );

  useEffect(() => {
    const setup = async () => {
      if (!session) {
        return;
      }
      const _ehm = await getHeatmapFromFile(
        session.texture,
        round,
        chosenPatchSize,
      );
      const _thm = await getHeatmapFromFile(session.texture, round, 1);
      setEhm(_ehm);
      setThm(_thm);
      setRandomFocusedTiles();
      setLoading(false);
    };
    setup();
  }, []);

  useEffect(() => {
    if (choiceCount === NUM_CHOICES) {
      navigate(round === NUM_ROUNDS ? "/complete" : "/main/round-complete");
    }
  }, [choiceCount, round, navigate]);

  return (
    <Box className="page">
      <TopBar />
      <Box direction="row" align="flex-start" className="grids-container">
        {loading ? (
          <span className="loading-text">Loading...</span>
        ) : (
          <>
            <Box className="map-container">
              <Box
                direction="row"
                justify="space-between"
                className="map-title"
              >
                <span>Map</span>
                {VEGETABLES[round - 1]}
              </Box>
              <TileGrid
                heatmap={ehm}
                dynamic={false}
                tileSize={200 / (8 / chosenPatchSize)}
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
                tileSize={50}
                tileMargin={2.5}
                tileRadius={7}
                dynamic={true}
                revealValues={false}
                recordChoices={true}
              />
            </Box>
          </>
        )}
      </Box>
    </Box>
  );
};

export default ChoicePage;
