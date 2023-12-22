import React, { useEffect } from "react";

import { useStore } from "../store";
import { GRID_SIZE } from "../constants";
import { getRandomHeatmap, getHeatmapFromFile } from "../api";
import { TileGrid } from "../components/TileGrid/TileGrid";

const EvidencePage = (): JSX.Element => {
  const heatmap = useStore((state) => state.heatmap);
  const setHeatmap = useStore((state) => state.setHeatmap);

  console.log(heatmap[0]);

  useEffect(() => {
    const setup = async () => {
      //   const hmap = await getRandomHeatmap();
      const hmap = await getHeatmapFromFile();
      setHeatmap(hmap);
    };
    setup();
  }, []);

  return (
    <div className="page">
      <TileGrid
        rows={GRID_SIZE}
        cols={GRID_SIZE}
        dynamic={false}
        heatmap={heatmap}
      />
    </div>
  );
};

export default EvidencePage;
