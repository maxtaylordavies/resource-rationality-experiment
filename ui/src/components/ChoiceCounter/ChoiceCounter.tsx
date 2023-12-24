import React from "react";

import { useStore } from "../../store";
import "./choice-counter.css";

type Props = {
  numChoices: number;
};

export const ChoiceCounter = ({ numChoices }: Props): JSX.Element => {
  const choiceCount = useStore((state) => state.choiceCount);

  return (
    <div className="choice-count-box">
      <span>Completed:</span>
      <span className="choice-count">
        {choiceCount}/{numChoices}
      </span>
    </div>
  );
};
