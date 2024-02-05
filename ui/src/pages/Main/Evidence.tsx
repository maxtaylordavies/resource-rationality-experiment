import React, { useEffect } from "react";

import { useStore } from "../../store";
import { getHeatmapFromFile } from "../../api";
import { Box } from "../../components/Box/Box";
import { TopBar } from "../../components/TopBar/TopBar";
import { TileGrid } from "../../components/TileGrid/TileGrid";
import { CountdownLink } from "../../components/CountdownLink/CountdownLink";

const EvidencePage = (): JSX.Element => {
  const [heatmap, setHeatmap] = useStore((state) => [
    state.heatmap,
    state.setHeatmap,
  ]);
  const round = useStore((state) => state.round);
  const chosenPatchSize = useStore((state) => state.chosenPatchSize);

  useEffect(() => {
    const setup = async () => {
      const hmap = await getHeatmapFromFile(round, chosenPatchSize);
      setHeatmap(hmap);
    };
    setup();
  }, [round, chosenPatchSize]);

  return (
    <Box className="page">
      <TopBar />
      <div className="evidence-grid-container">
        <img
          src={window.location.origin + "/assets/key-potato.png"}
          className="evidence-key-image"
        />
        <TileGrid
          heatmap={heatmap}
          dynamic={false}
          tileSize={50 * chosenPatchSize}
          tileMargin={2 * chosenPatchSize}
          tileRadius={5 * chosenPatchSize}
        />
      </div>
      <CountdownLink
        to="/main/choice"
        duration={10}
        className="corner-countdown-link"
      />
    </Box>
  );
};

export default EvidencePage;
