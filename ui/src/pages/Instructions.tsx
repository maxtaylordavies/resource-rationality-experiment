import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";

import { useStore } from "../store";
import { createSession } from "../api";
import { getValueFromUrlOrLocalstorage } from "../utils";

type ContentItem = {
  text: string;
  imgUrl: string;
};

const content: ContentItem[][] = [
  [
    {
      text: "You are going to play a short game. In this game, you have been hired to assist a local farmer.",
      imgUrl: "farmer.png",
    },
    {
      text: "The farmer has a field. The field is divided equally into a grid of 64 square plots.",
      imgUrl: "grid.png",
    },
  ],
  [
    {
      text: "In each round of the game, you will help the farmer grow a particular vegetable crop.",
      imgUrl: "vegetables.png",
    },
    {
      text: "Depending on the type of crop, different plots are more or less suitable for planting.",
      imgUrl: "grid-tick-cross.png",
    },
    {
      text: "In particular, plots are divided into four levels of quality, indicated by red, orange, yellow and green.",
      imgUrl: "colours.png",
    },
  ],
  [
    {
      text: "You will be shown pairs of plots in the field. For each pair, choose which plot the farmer should plant in.",
      imgUrl: "two-plots.png",
    },
    {
      text: "The farmer will reward you with points for correct choices. Incorrect choices will earn no points.",
      imgUrl: "moneybag.png",
    },
    {
      text: "You will get to see a map of the plot colours before each round, but not while you make choices.",
      imgUrl: "colour-grid.png",
    },
  ],
];

const InstructionsPage = (): JSX.Element => {
  const navigate = useNavigate();

  const session = useStore((state) => state.session);
  const setSession = useStore((state) => state.setSession);

  const [page, setPage] = useState(0);

  useEffect(() => {
    const setup = async () => {
      const session = await createSession(
        getValueFromUrlOrLocalstorage("expId"),
        getValueFromUrlOrLocalstorage("userId")
      );
      setSession(session);
    };
    setup();
  }, []);

  const onNextClick = () => {
    if (page < content.length - 1) {
      setPage(page + 1);
    } else {
      navigate("/evidence1");
    }
  };

  return (
    <div className="instructions-page">
      <h1>Instructions</h1>
      {content[page].map((item, idx) => (
        <div key={idx} className="box">
          <span>{item.text}</span>
          <img src={window.location.origin + "/assets/" + item.imgUrl} />
        </div>
      ))}
      <div className="button-container">
        {page > 0 && (
          <motion.button
            className="back-button"
            onClick={() => setPage(page - 1)}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            Back
          </motion.button>
        )}
        <motion.button
          className="next-button"
          onClick={onNextClick}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          Next
        </motion.button>
      </div>
    </div>
  );
};

export default InstructionsPage;
