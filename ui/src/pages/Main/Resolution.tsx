import { useState } from "react";
import { motion } from "framer-motion";

import { useStore } from "../../store";
import { PATCH_SIZES, MAP_COSTS } from "../../constants";
import { Box } from "../../components/Box/Box";
import { TopBar } from "../../components/TopBar/TopBar";
import { Button } from "../../components/Button/Button";
import { LinkButton } from "../../components/Button/LinkButton";

const ResolutionPage = (): JSX.Element => {
  const [chosenPatchSize, setChosenPatchSize] = useStore((state) => [
    state.chosenPatchSize,
    state.setChosenPatchSize,
  ]);

  const [hoverIdx, setHoverIdx] = useState<number>(-1);

  return (
    <Box className="page">
      <TopBar />
      <h1 className="resolution-page-title">Choose a map to purchase</h1>
      <div className="resolution-page-options">
        {MAP_COSTS.map((cost, idx) => {
          const selected = chosenPatchSize === PATCH_SIZES[idx];
          return (
            <div
              className={`resolution-page-option${selected ? " selected" : ""}`}
              key={idx}
            >
              <motion.img
                src={`${window.location.origin}/assets/resolution-${idx}.png`}
                alt=""
                initial={{ opacity: 0.15, scale: 1.0 }}
                animate={{
                  opacity: selected ? 1.0 : 0.15,
                  scale: selected || hoverIdx === idx ? 1.05 : 1.0,
                }}
              />
              <Button
                label={`${cost} points`}
                onClick={() => setChosenPatchSize(PATCH_SIZES[idx])}
                variant="primary"
                onMouseEnter={() => setHoverIdx(idx)}
                onMouseLeave={() => setHoverIdx(-1)}
                animate={{ scale: selected ? 1.05 : 1.0 }}
              />
            </div>
          );
        })}
      </div>
      <LinkButton
        label="Confirm"
        to="/main/evidence"
        variant="primary"
        style={{
          position: "absolute",
          bottom: 35,
          right: 35,
          pointerEvents: chosenPatchSize === -1 ? "none" : "auto",
          opacity: chosenPatchSize === -1 ? 0.5 : 1.0,
        }}
      />
    </Box>
  );
};

export default ResolutionPage;
