import { useState } from "react";
import { useNavigate } from "react-router-dom";

import { Box } from "../components/Box/Box";
import { Button } from "../components/Button/Button";
// import { Session, useStore } from "../store";
// import { getSession, createSession } from "../api";
// import { getValueFromUrlOrLocalstorage, writeToLocalStorage } from "../utils";

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
  // const setSession = useStore((state) => state.setSession);
  const [page, setPage] = useState(0);

  // useEffect(() => {
  //   const setup = async () => {
  //     let sess;

  //     // check for existing session
  //     const sid = getValueFromUrlOrLocalstorage("sessionid");
  //     if (sid) {
  //       sess = await getSession(sid);
  //     }

  //     // create new session if none exists
  //     if (!sess) {
  //       sess = await createSession(
  //         getValueFromUrlOrLocalstorage("expid"),
  //         getValueFromUrlOrLocalstorage("userid"),
  //       );
  //     }

  //     // save session id to localstorage and save session to global store
  //     writeToLocalStorage("sessionid", sess.id);
  //     setSession(sess);
  //   };
  //   setup();
  // }, []);

  const onNextClick = () => {
    if (page < content.length - 1) {
      setPage(page + 1);
    } else {
      navigate("/tutorial/start");
    }
  };

  return (
    <Box className="instructions-page">
      <h1>Instructions</h1>
      {content[page].map((item) => (
        <Box key={item.text} className="box">
          <span>{item.text}</span>
          <img src={window.location.origin + "/assets/" + item.imgUrl} />
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
