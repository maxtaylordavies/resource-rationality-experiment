import React, { useEffect } from "react";

import { useStore } from "../store";
import { createSession } from "../api";
import { getValueFromUrlOrLocalstorage } from "../utils";
import { LinkButton } from "../components/Button/LinkButton";

const InstructionsPage = (): JSX.Element => {
  const session = useStore((state) => state.session);
  const setSession = useStore((state) => state.setSession);

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

  return (
    <div className="page">
      <h1>Instructions</h1>
      {session !== null && (
        <LinkButton to="/evidence1" label="Proceed" style={{ marginTop: 10 }} />
      )}
    </div>
  );
};

export default InstructionsPage;
