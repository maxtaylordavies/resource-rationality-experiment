import React, { useState } from "react";
import { motion } from "framer-motion";

import { useStore } from "../../store";
import { MAP_COSTS } from "../../constants";
import { Box } from "../../components/Box/Box";
import { Button } from "../../components/Button/Button";
import { LinkButton } from "../../components/Button/LinkButton";

const ResolutionPage = (): JSX.Element => {
  const [resolutionChoice, setResolutionChoice] = useStore((state) => [
    state.resolutionChoice,
    state.setResolutionChoice,
  ]);

  const [hoverIdx, setHoverIdx] = useState<number>(-1);

  return (
    <Box className="page">
      <h1 className="resolution-page-title">Choose map resolution</h1>
      <div className="resolution-page-options">
        {MAP_COSTS.map((cost, idx) => {
          const selected = resolutionChoice === idx;
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
                onClick={() => setResolutionChoice(idx)}
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
          pointerEvents: resolutionChoice === -1 ? "none" : "auto",
          opacity: resolutionChoice === -1 ? 0.5 : 1.0,
        }}
      />
    </Box>
  );
};

export default ResolutionPage;
