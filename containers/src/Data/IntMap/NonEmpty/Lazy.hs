{-# LANGUAGE CPP #-}
#if !defined(TESTING) && defined(__GLASGOW_HASKELL__)
{-# LANGUAGE Safe #-}
#endif

#include "containers.h"

-----------------------------------------------------------------------------
-- |
-- Module      :  Data.IntMap.NonEmpty.Lazy
-- License     :  BSD-style
-- Maintainer  :  libraries@haskell.org
-- Portability :  portable
--
--
-- = Non-empty Finite Int Maps (lazy interface)
--
-- The @'IntMap' v@ type represents a finite map (sometimes called a dictionary)
-- from keys of type @Int@ to values of type @v@.
--
-- The functions in "Data.IntMap.NonEmpty.Strict" are careful to force values before
-- installing them in an 'IntMap'. This is usually more efficient in cases where
-- laziness is not essential. The functions in this module do not do so.
--
-- For a walkthrough of the most commonly used functions see the
-- <https://haskell-containers.readthedocs.io/en/latest/map.html maps introduction>.
--
-- This module is intended to be imported qualified, to avoid name clashes with
-- Prelude functions:
--
-- > import Data.IntMap.NonEmpty.Lazy (IntMap)
-- > import qualified Data.IntMap.NonEmpty.Lazy as IntMap
--
-- Note that the implementation is generally /left-biased/. Functions that take
-- two maps as arguments and combine them, such as `union` and `intersection`,
-- prefer the values in the first argument to those in the second.
--
--
-- == Detailed performance information
--
-- The amortized running time is given for each operation, with \(n\) referring to
-- the number of entries in the map and \(W\) referring to the number of bits in
-- an 'Int' (32 or 64).
--
-- Benchmarks comparing "Data.IntMap.Lazy" with other dictionary
-- implementations can be found at https://github.com/haskell-perf/dictionaries.
--
--
-- == Implementation
--
-- The implementation is based on /big-endian patricia trees/.  This data
-- structure performs especially well on binary operations like 'union' and
-- 'intersection'. Additionally, benchmarks show that it is also (much) faster
-- on insertions and deletions when compared to a generic size-balanced map
-- implementation (see "Data.Map").
--
--    * Chris Okasaki and Andy Gill,  \"/Fast Mergeable Integer Maps/\",
--      Workshop on ML, September 1998, pages 77-86,
--      <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.37.5452>
--
--    * D.R. Morrison, \"/PATRICIA -- Practical Algorithm To Retrieve Information Coded In Alphanumeric/\",
--      Journal of the ACM, 15(4), October 1968, pages 514-534.
--
-----------------------------------------------------------------------------

module Data.IntMap.NonEmpty.Lazy (
    -- * Map type
#if !defined(TESTING)
    IntMap, Key          -- instance Eq,Show
#else
    IntMap(..), Key          -- instance Eq,Show
#endif

    -- * Construction
    , singleton

    -- ** From Unordered Lists
    , fromList
    , fromListWith
    , fromListWithKey

    -- ** From Ascending Lists
    , fromAscList
    , fromAscListWith
    , fromAscListWithKey
    , fromDistinctAscList

    -- * Insertion
    , insert
    , insertWith
    , insertWithKey
    , insertLookupWithKey

    -- * Deletion\/Update
    , adjust
    , adjustWithKey

    -- * Query
    -- ** Lookup
    , IM.lookup
    , (!?)
    , (!)
    , findWithDefault
    , member
    , notMember
    , lookupLT
    , lookupGT
    , lookupLE
    , lookupGE

    -- ** Size
    , size

    -- * Combine

    -- ** Union
    , union
    , unionWith
    , unionWithKey

    -- ** Disjoint
    , disjoint

    -- ** Universal combining function
    , mergeWithKey

    -- * Traversal
    -- ** Map
    , IM.map
    , mapWithKey
    , traverseWithKey
    , mapAccum
    , mapAccumWithKey
    , mapAccumRWithKey
    , mapKeys
    , mapKeysWith
    , mapKeysMonotonic

    -- * Folds
    , IM.foldr
    , IM.foldl
    , foldrWithKey
    , foldlWithKey
    , foldMapWithKey

    -- ** Strict folds
    , foldr'
    , foldl'
    , foldrWithKey'
    , foldlWithKey'

    -- * Conversion
    , elems
    , keys
    , assocs
    , keysSet

    -- ** Lists
    , toList

    -- ** Ordered lists
    , toAscList
    , toDescList

    -- * Split
    , splitRoot

    -- * Submap
    , isSubmapOf, isSubmapOfBy
    , isProperSubmapOf, isProperSubmapOfBy

    -- * Min\/Max
    , findMin
    , findMax
    , updateMinWithKey
    , updateMaxWithKey

    ) where

import Data.IntMap.NonEmpty.Internal as IM
