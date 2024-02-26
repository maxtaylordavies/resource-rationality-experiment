import { useNavigate } from "react-router-dom";
import { HTMLMotionProps, motion } from "framer-motion";

import "./button.css";

// extend motion button props
interface Props extends HTMLMotionProps<"button"> {
  label: string;
  to: string;
  variant: "primary" | "secondary";
  onClick?: () => void;
}

export const LinkButton = ({
  label,
  to,
  variant,
  onClick,
  ...props
}: Props) => {
  const navigate = useNavigate();

  const handleClick = () => {
    if (onClick) {
      onClick();
    }
    navigate(to);
  };

  return (
    <motion.button
      className={`button ${variant}`}
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      onClick={handleClick}
      {...props}
    >
      {label}
    </motion.button>
  );
};
