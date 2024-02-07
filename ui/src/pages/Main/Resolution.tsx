import { useState } from "react";
import { motion } from "framer-motion";

import { useStore } from "../../store";
import { PATCH_SIZES, MAP_COSTS } from "../../constants";
import { Box } from "../../components/Box/Box";
import { TopBar } from "../../components/TopBar/TopBar";
import { Button } from "../../components/Button/Button";
import { LinkButton } from "../../components/Button/LinkButton";
import { Coin } from "../../components/Coin/Coin";

const ResolutionPage = (): JSX.Element => {
  const setChosenPatchSize = useStore((state) => state.setChosenPatchSize);
  const incrementScore = useStore((state) => state.incrementScore);

  const [hoverIdx, setHoverIdx] = useState<number>(-1);
  const [selectedIdx, setSelectedIdx] = useState<number>(-1);

  return (
    <Box className="page">
      <TopBar />
      <h1 className="resolution-page-title">Choose a map to purchase</h1>
      <div className="resolution-page-options">
        {MAP_COSTS.map((cost, idx) => {
          const selected = selectedIdx === idx;
          return (
            <div
              className={`resolution-page-option${selected ? " selected" : ""}`}
              key={idx}
            >
              <motion.img
                className="resolution-page-option-image"
                src={`${window.location.origin}/assets/resolution-${idx}.png`}
                alt=""
                initial={{ opacity: 0.15, scale: 1.0 }}
                animate={{
                  opacity: selected ? 1.0 : 0.15,
                  scale: selected || hoverIdx === idx ? 1.05 : 1.0,
                }}
              />
              <Button
                label=""
                onClick={() => setSelectedIdx(idx)}
                variant="primary"
                onMouseEnter={() => setHoverIdx(idx)}
                onMouseLeave={() => setHoverIdx(-1)}
                animate={{ scale: selected ? 1.05 : 1.0 }}
              >
                {cost} <Coin height={28} style={{ marginLeft: 5 }} />
              </Button>
            </div>
          );
        })}
      </div>
      <LinkButton
        onClick={() => {
          setChosenPatchSize(PATCH_SIZES[selectedIdx]);
          const delta = -MAP_COSTS[selectedIdx];
          console.log("incrementing score by", delta);
          incrementScore(delta);
        }}
        label="Confirm"
        to="/main/choice"
        variant="primary"
        style={{
          position: "absolute",
          bottom: 35,
          right: 35,
          pointerEvents: selectedIdx === -1 ? "none" : "auto",
          opacity: selectedIdx === -1 ? 0.5 : 1.0,
        }}
      />
    </Box>
  );
};

export default ResolutionPage;
