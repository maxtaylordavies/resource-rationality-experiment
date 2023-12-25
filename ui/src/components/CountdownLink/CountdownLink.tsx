import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { CountdownCircleTimer } from "react-countdown-circle-timer";

type Props = {
  duration: number;
  to: string;
};

export const CountdownLink = ({ duration, to }: Props) => {
  const navigate = useNavigate();
  const [timerKey, setTimerKey] = useState<number>(0);

  return (
    <CountdownCircleTimer
      key={timerKey}
      isPlaying
      duration={duration}
      onComplete={(totalElapsedTime) => {
        setTimerKey(timerKey + 1);
        navigate(to);
      }}
      colors={"#808080"}
      trailColor="#d8d8d8"
      size={100}
    >
      {({ remainingTime }) => (
        <span style={{ fontSize: 32, fontWeight: 600, color: "#808080" }}>
          {remainingTime}
        </span>
      )}
    </CountdownCircleTimer>
  );
};
