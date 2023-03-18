module IntMapValidity (valid) where

import Data.Bits (xor, (.&.))
import Data.IntMap.Internal
import qualified Data.IntMap.NonEmpty.Internal as NE
import Test.Tasty.QuickCheck (Property, counterexample, property, (.&&.))
import Utils.Containers.Internal.BitUtil (bitcount)

{--------------------------------------------------------------------
  Assertions
--------------------------------------------------------------------}
-- | Returns true iff the internal structure of the IntMap is valid.
valid :: IntMap a -> Property
valid t =
  counterexample "commonPrefix" (commonPrefix t) .&&.
  counterexample "maskRespected" (maskRespected t)

-- Invariant: The Mask is a power of 2. It is the largest bit position at which
--            two keys of the map differ.
maskPowerOfTwo :: IntMap a -> Bool
maskPowerOfTwo Nil = True
maskPowerOfTwo (NonEmpty t0) = go t0
  where go (NE.Tip {}) = True
        go (NE.Bin _ m l r) =
          bitcount 0 (fromIntegral m) == 1 && go l && go r

-- Invariant: Prefix is the common high-order bits that all elements share to
--            the left of the Mask bit.
commonPrefix :: IntMap a -> Bool
commonPrefix Nil = True
commonPrefix (NonEmpty t0) = go t0
  where
    go (NE.Tip {}) = True
    go b@(NE.Bin p _ l r) =
      all (sharedPrefix p) (NE.keys b) && go l && go r
    sharedPrefix :: Prefix -> Int -> Bool
    sharedPrefix p a = p == p .&. a

-- Invariant: In Bin prefix mask left right, left consists of the elements that
--            don't have the mask bit set; right is all the elements that do.
maskRespected :: IntMap a -> Bool
maskRespected Nil = True
maskRespected (NonEmpty t0) = go t0
  where
    go (NE.Tip {}) = True
    go (NE.Bin _ binMask l r) =
       all (\x -> zero x binMask) (NE.keys l) &&
       all (\x -> not (zero x binMask)) (NE.keys r) &&
       go l &&
       go r
