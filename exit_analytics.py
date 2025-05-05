"""
Exit analytics module for tracking and analyzing position exits.
"""

import logging
from datetime import datetime

logger = logging.getLogger("BinanceTrading.ExitAnalytics")

class ExitAnalytics:
    """Analyze position exits to improve strategy"""
    
    def __init__(self):
        self.exit_reasons = {
            'stop_loss': 0,
            'take_profit': 0,
            'position_aging': 0,
            'market_regime_change': 0,
            'manual': 0,
            'risk_management': 0
        }
        self.profitable_exits = {key: 0 for key in self.exit_reasons}
        self.total_pnl_by_reason = {key: 0 for key in self.exit_reasons}
        
    def record_exit(self, position):
        """Record an exit and update statistics"""
        exit_reason = position.get('exit_reason', 'unknown')
        
        # Increment exit count
        if exit_reason in self.exit_reasons:
            self.exit_reasons[exit_reason] += 1
            
            # Track if exit was profitable
            net_pnl = position.get('net_pnl', 0)
            if net_pnl > 0:
                self.profitable_exits[exit_reason] += 1
                
            # Track total P&L by reason
            self.total_pnl_by_reason[exit_reason] += net_pnl
        else:
            # Handle unknown exit reasons
            if 'unknown' not in self.exit_reasons:
                self.exit_reasons['unknown'] = 0
                self.profitable_exits['unknown'] = 0
                self.total_pnl_by_reason['unknown'] = 0
            
            self.exit_reasons['unknown'] += 1
            if position.get('net_pnl', 0) > 0:
                self.profitable_exits['unknown'] += 1
            self.total_pnl_by_reason['unknown'] += position.get('net_pnl', 0)
            
    def get_exit_statistics(self):
        """Get statistics on position exits"""
        stats = {}
        total_exits = sum(self.exit_reasons.values())
        
        if total_exits > 0:
            stats['exit_distribution'] = {
                reason: {
                    'count': count,
                    'percentage': (count / total_exits) * 100 if total_exits > 0 else 0
                }
                for reason, count in self.exit_reasons.items()
            }
            
            stats['win_rate_by_reason'] = {
                reason: (self.profitable_exits[reason] / count) * 100 if count > 0 else 0
                for reason, count in self.exit_reasons.items()
            }
            
            stats['avg_pnl_by_reason'] = {
                reason: self.total_pnl_by_reason[reason] / count if count > 0 else 0
                for reason, count in self.exit_reasons.items()
            }
            
        return stats
    
    def save_to_dict(self):
        """Convert analytics to dictionary for saving to state file"""
        return {
            'exit_reasons': self.exit_reasons,
            'profitable_exits': self.profitable_exits,
            'total_pnl_by_reason': self.total_pnl_by_reason
        }
    
    def load_from_dict(self, data):
        """Load analytics from dictionary (from state file)"""
        if data:
            self.exit_reasons.update(data.get('exit_reasons', {}))
            self.profitable_exits.update(data.get('profitable_exits', {}))
            self.total_pnl_by_reason.update(data.get('total_pnl_by_reason', {}))