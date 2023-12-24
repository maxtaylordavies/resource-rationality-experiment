import React, { useEffect } from "react";

import { useStore } from "../store";
import { getHeatmapFromFile } from "../api";
import { TileGrid } from "../components/TileGrid/TileGrid";
import { LinkButton } from "../components/Button/LinkButton";

const EvidencePage = (): JSX.Element => {
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
      <TileGrid dynamic={false} tileSize={60} />
      <LinkButton to="/test1" label="Proceed" style={{ marginTop: 10 }} />
    </div>
  );
};

export default EvidencePage;
