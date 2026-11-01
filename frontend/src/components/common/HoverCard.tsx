import { cloneElement, useEffect, useId, useRef, useState, type ReactElement } from 'react';
import { createPortal } from 'react-dom';

const OPEN_DELAY_MS = 250;
const GAP_PX = 10;

interface HoverCardProps {
  title: string;
  text: string;
  colour: string;
  children: ReactElement<{ 'aria-describedby'?: string }>;
}

/** A short card to the right of its child. Portalled, because the side panel clips overflow. */
export function HoverCard({ title, text, colour, children }: HoverCardProps) {
  const id = useId();
  const anchor = useRef<HTMLDivElement>(null);
  const timer = useRef<number | undefined>(undefined);
  const [position, setPosition] = useState<{ left: number; top: number } | null>(null);

  const show = () => {
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => {
      const rect = anchor.current?.getBoundingClientRect();
      if (rect) setPosition({ left: rect.right + GAP_PX, top: rect.top + rect.height / 2 });
    }, OPEN_DELAY_MS);
  };

  const hide = () => {
    window.clearTimeout(timer.current);
    setPosition(null);
  };

  useEffect(() => () => window.clearTimeout(timer.current), []);

  const isOpen = position !== null;
  useEffect(() => {
    if (!isOpen) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setPosition(null);
    };
    const onScroll = () => setPosition(null);
    window.addEventListener('keydown', onKey);
    window.addEventListener('scroll', onScroll, true);
    return () => {
      window.removeEventListener('keydown', onKey);
      window.removeEventListener('scroll', onScroll, true);
    };
  }, [isOpen]);

  return (
    <div ref={anchor} className="hover-anchor" onMouseEnter={show} onMouseLeave={hide} onFocus={show} onBlur={hide}>
      {cloneElement(children, { 'aria-describedby': isOpen ? id : undefined })}
      {position &&
        createPortal(
          <div
            id={id}
            role="tooltip"
            className="hover-card-tip"
            style={{ left: position.left, top: position.top, ['--hover-colour' as string]: colour }}
          >
            <strong>{title}</strong>
            {text}
          </div>,
          document.body,
        )}
    </div>
  );
}
