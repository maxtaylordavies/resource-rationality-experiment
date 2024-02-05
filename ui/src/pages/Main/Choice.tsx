import { useEffect } from "react";
import { useNavigate } from "react-router-dom";

import { useStore } from "../../store";
import { NUM_ROUNDS, NUM_CHOICES } from "../../constants";
import { getHeatmapFromFile } from "../../api";
import { Box } from "../../components/Box/Box";
import { TileGrid } from "../../components/TileGrid/TileGrid";
import { TopBar } from "../../components/TopBar/TopBar";

const ChoicePage = (): JSX.Element => {
  const navigate = useNavigate();

  const round = useStore((state) => state.round);
  const [heatmap, setHeatmap] = useStore((state) => [
    state.heatmap,
    state.setHeatmap,
  ]);
  const choiceCount = useStore((state) => state.choiceCount);
  const setRandomFocusedTiles = useStore(
    (state) => state.setRandomFocusedTiles,
  );

  useEffect(() => {
    const setup = async () => {
      const hmap = await getHeatmapFromFile(round, 1);
      setHeatmap(hmap);
      setRandomFocusedTiles();
    };
    setup();
  }, []);

  useEffect(() => {
    if (choiceCount === NUM_CHOICES) {
      navigate(round === NUM_ROUNDS ? "/complete" : "/main/round-complete");
    }
  }, [choiceCount]);

  return (
    <Box className="page">
      <TopBar />
      <div className="choice-grid-container">
        <img
          src={window.location.origin + "/assets/which-plot.png"}
          className="which-choice-image"
          alt=""
        />
        <TileGrid
          heatmap={heatmap}
          tileSize={50}
          tileMargin={2}
          tileRadius={5}
          dynamic={true}
          revealValues={false}
          recordChoices={true}
        />
      </div>
    </Box>
  );
};

export default ChoicePage;
