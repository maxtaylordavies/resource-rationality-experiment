import React, { useEffect } from "react";

import { useStore } from "../../store";
import { getHeatmapFromFile } from "../../api";
import { TileGrid } from "../../components/TileGrid/TileGrid";
import { CountdownLink } from "../../components/CountdownLink/CountdownLink";

const EvidencePage = (): JSX.Element => {
  const heatmap = useStore((state) => state.heatmap);
  const setHeatmap = useStore((state) => state.setHeatmap);

  useEffect(() => {
    const setup = async () => {
      const hmap = await getHeatmapFromFile("1");
      setHeatmap(hmap);
    };
    setup();
  }, []);

  return (
    <div className="page">
      <img
        src={window.location.origin + "/assets/robot-scale.png"}
        className="robot-scale-image"
      />
      <TileGrid heatmap={heatmap} dynamic={false} tileSize={60} />
      <CountdownLink to="/test1" duration={10} />
    </div>
  );
};

export default EvidencePage;
