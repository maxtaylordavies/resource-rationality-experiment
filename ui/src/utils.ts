const PREFIX = "_rre_";

export const getValueFromUrlOrLocalstorage = (key: string) => {
  const params = new URLSearchParams(window.location.search);
  return (
    params.get(key) ||
    JSON.parse(localStorage.getItem(`${PREFIX}${key}`) || "null")
  );
};

export const writeToLocalStorage = (key: string, val: any) => {
  localStorage.setItem(`${PREFIX}${key}`, JSON.stringify(val));
};

export const removeFromLocalStorage = (key: string) => {
  localStorage.removeItem(`${PREFIX}${key}`);
};
