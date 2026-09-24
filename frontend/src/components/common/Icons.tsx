import type { SVGProps } from 'react';

const base = {
  viewBox: '0 0 16 16',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.5,
  strokeLinecap: 'round' as const,
  strokeLinejoin: 'round' as const,
  'aria-hidden': true,
};

type IconProps = SVGProps<SVGSVGElement>;

export const PlayIcon = (p: IconProps) => (
  <svg {...base} {...p}>
    <path d="M4.5 3v10l8-5z" fill="currentColor" stroke="none" />
  </svg>
);

export const ResetIcon = (p: IconProps) => (
  <svg {...base} {...p}>
    <path d="M3 8a5 5 0 1 0 1.5-3.6M3 2.5v2.5h2.5" />
  </svg>
);

export const CameraIcon = (p: IconProps) => (
  <svg {...base} {...p}>
    <path d="M2 8a6 6 0 0 1 12 0M2 8a6 6 0 0 0 12 0M8 2v2M8 12v2M5.5 8h5" />
  </svg>
);

export const MapIcon = (p: IconProps) => (
  <svg {...base} {...p}>
    <path d="M1.5 3.5l4-1.5 5 1.5 4-1.5v10.5l-4 1.5-5-1.5-4 1.5zM5.5 2v10.5M10.5 3.5V14" />
  </svg>
);

export const PinIcon = (p: IconProps) => (
  <svg {...base} {...p}>
    <path d="M8 14.5s4.5-4.2 4.5-7.8a4.5 4.5 0 0 0-9 0c0 3.6 4.5 7.8 4.5 7.8z" />
    <circle cx="8" cy="6.5" r="1.6" />
  </svg>
);

export const DiceIcon = (p: IconProps) => (
  <svg {...base} {...p}>
    <rect x="2.5" y="2.5" width="11" height="11" rx="2" />
    <circle cx="5.6" cy="5.6" r=".8" fill="currentColor" />
    <circle cx="10.4" cy="10.4" r=".8" fill="currentColor" />
    <circle cx="8" cy="8" r=".8" fill="currentColor" />
  </svg>
);

export const TargetIcon = (p: IconProps) => (
  <svg {...base} {...p}>
    <circle cx="8" cy="8" r="5.5" />
    <circle cx="8" cy="8" r="2" />
  </svg>
);

export const OverviewIcon = (p: IconProps) => (
  <svg {...base} {...p}>
    <path d="M1.5 11.5L8 14l6.5-2.5M1.5 8L8 10.5 14.5 8M8 2L1.5 4.5 8 7l6.5-2.5z" />
  </svg>
);

export const InfoIcon = (p: IconProps) => (
  <svg {...base} {...p}>
    <circle cx="8" cy="8" r="6" />
    <path d="M8 7.2v4M8 4.8v.01" />
  </svg>
);

export const WarnIcon = (p: IconProps) => (
  <svg {...base} {...p}>
    <path d="M8 2l6.5 11.5h-13zM8 6.5v3.2M8 11.8v.01" />
  </svg>
);

export const CheckIcon = (p: IconProps) => (
  <svg {...base} {...p}>
    <path d="M3 8.5l3 3 7-7" />
  </svg>
);

export const CrossIcon = (p: IconProps) => (
  <svg {...base} {...p}>
    <path d="M4 4l8 8M12 4l-8 8" />
  </svg>
);

export const FlagIcon = (p: IconProps) => (
  <svg {...base} {...p}>
    <path d="M3.5 14.5V2M3.5 2.5h8l-1.8 3 1.8 3h-8" />
  </svg>
);

export const SearchIcon = (p: IconProps) => (
  <svg {...base} {...p}>
    <circle cx="7" cy="7" r="4.5" />
    <path d="M10.5 10.5L14 14" />
  </svg>
);

export const EyeIcon = (p: IconProps) => (
  <svg {...base} {...p}>
    <path d="M1.5 8S4 3.5 8 3.5 14.5 8 14.5 8 12 12.5 8 12.5 1.5 8 1.5 8z" />
    <circle cx="8" cy="8" r="2" />
  </svg>
);

export const LayersIcon = OverviewIcon;

export const HelpIcon = (p: IconProps) => (
  <svg {...base} {...p}>
    <circle cx="8" cy="8" r="6" />
    <path d="M6.3 6.2a1.8 1.8 0 1 1 2.4 1.7c-.5.2-.7.6-.7 1.1v.3M8 11.6v.01" />
  </svg>
);

export const MountainMark = (p: IconProps) => (
  <svg viewBox="0 0 32 32" aria-hidden="true" {...p}>
    <rect x="1" y="1" width="30" height="30" rx="7" fill="#161b20" stroke="#2e363e" />
    <path d="M5 24 L13 11 L17.5 17.5 L20.5 13.5 L27 24 Z" fill="#3f4a55" />
    <path d="M13 11 L17.5 17.5 L15.2 21 L10.5 15 Z" fill="#d9793a" />
    <circle cx="15.3" cy="16" r="1.4" fill="#e3e7eb" />
  </svg>
);
