import { useState } from "react";
import { useNavigate } from "react-router-dom";

import { Box } from "../components/Box/Box";
import { Button } from "../components/Button/Button";

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
      text: "The farmer has a field. The field is divided equally into a grid of square plots.",
      imgUrl: "grid.png",
    },
  ],
  [
    {
      text: "In each round of the game, you will help the farmer grow a particular vegetable crop.",
      imgUrl: "vegetables.png",
    },
    {
      text: "You will be shown pairs of plots in the field. For each pair, choose which plot the farmer should plant in.",
      imgUrl: "two-plots.png",
    },
  ],
  [
    {
      text: "Depending on the type of crop, different plots are more or less suitable for planting.",
      imgUrl: "grid-tick-cross.png",
    },
    {
      text: "You will have a map showing how suitable different plots are. Green plots are the best, red are the worst.",
      imgUrl: "colour-grid.png",
    },
  ],
  [
    {
      text: "You will earn coins for each correct choice, and lose coins for each incorrect choice.",
      imgUrl: "coin.png",
    },
    {
      text: "Your goal is to earn as many coins as possible. If you earn enough coins, you will receive a bonus payment!",
      imgUrl: "moneybag.png",
    },
  ],
];

const InstructionsPage = (): JSX.Element => {
  const navigate = useNavigate();
  const [page, setPage] = useState(0);

  const onNextClick = () => {
    if (page < content.length - 1) {
      setPage(page + 1);
    } else {
      navigate("/tutorial/start");
    }
  };

  return (
    <Box align="flex-start" className="instructions-page">
      <h1>Instructions</h1>
      {content[page].map((item) => (
        <Box
          key={item.text}
          direction="row"
          justify="space-between"
          className="box"
        >
          <span>{item.text}</span>
          <img
            src={window.location.origin + "/assets/" + item.imgUrl}
            alt={item.imgUrl}
          />
        </Box>
      ))}
      <div className="button-container">
        {page > 0 && (
          <Button
            label="Back"
            onClick={() => setPage(page - 1)}
            variant="secondary"
            style={{ marginRight: 10 }}
          />
        )}
        <Button
          label={page === content.length - 1 ? "Begin" : "Next"}
          onClick={onNextClick}
          variant="primary"
        />
      </div>
    </Box>
  );
};

export default InstructionsPage;
