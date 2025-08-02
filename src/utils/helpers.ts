// utils/helpers.ts
export const getStatusColor = (status: string): string => {
  switch (status) {
    case 'completed': return 'bg-emerald-100 text-emerald-800';
    case 'failed': return 'bg-rose-100 text-rose-800';
    case 'processing': return 'bg-amber-100 text-amber-800';
    default: return 'bg-gray-100 text-gray-800';
  }
};

export const getSubscriptionBadgeClass = (tier: string): string => {
  const colors: Record<string, string> = {
    free: 'bg-stone-100 text-stone-700',
    premium: 'bg-amber-100 text-amber-700'
  };
  return colors[tier] || colors.free;
};
