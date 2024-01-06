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
      <div className="evidence-grid-container">
        <img
          src={window.location.origin + "/assets/key-potato.png"}
          className="evidence-key-image"
        />
        <TileGrid
          heatmap={heatmap}
          dynamic={false}
          tileSize={130}
          tileMargin={5}
          tileRadius={20}
        />
      </div>
      <CountdownLink
        to="/tutorial/choice"
        duration={10}
        className="corner-countdown-link"
      />
    </div>
  );
};

export default EvidencePage;
