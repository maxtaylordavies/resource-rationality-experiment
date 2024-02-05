import React from "react";

import { NUM_ROUNDS, NUM_CHOICES } from "../../constants";
import { useStore } from "../../store";
import "./top-bar.css";

export const TopBar = (): JSX.Element => {
  const round = useStore((state) => state.round);
  const choiceCount = useStore((state) => state.choiceCount);
  const score = useStore((state) => state.score);

  const roundText = window.location.pathname.includes("tutorial")
    ? "Tutorial"
    : `Round ${round} of ${NUM_ROUNDS}`;

  return (
    <div className="top-bar">
      <div className="round-text">{roundText}</div>
      <div className="score-box">
        <span className="score-box-score">{score}</span> points
      </div>
      <div className="choice-count-box">
        <span className="choice-count">
          {choiceCount}/{NUM_CHOICES}
        </span>
        <span>pairs done</span>
      </div>
    </div>
  );
};
